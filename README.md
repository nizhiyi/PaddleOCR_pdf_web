
# 基于 PaddleOCR 的高性能可搜索 PDF 生成服务：从多核加速到 Web 集成实战

## 1. 引言
在数字化办公场景中，将扫描件或纯图片 PDF 转换为“可搜索、可选择、可复制”的 PDF 是一项刚需。本文将分享一个基于 **PaddleOCR** 和 **FastAPI** 开发的高性能 PDF OCR 服务。该项目通过多进程并行处理和二分字号匹配算法，实现了生产级的处理速度与渲染精度。

---

## 2. 核心亮点
*   **🚀 多核并行加速**：利用 `ProcessPoolExecutor` 实现页面级的并行 OCR，极大提升了大文件处理速度。
*   **🎯 精准渲染对齐**：采用二分搜索算法自动匹配最佳字号，确保透明文本层与原图文字完美重合。
*   **📊 实时进度追踪**：通过 SSE（Server-Sent Events）技术，在前端实时展示“渲染-识别-合并”的全流程百分比进度。
*   **🐳 Docker 一键部署**：完整集成环境配置，支持在容器内快速启动服务。
*   **💾 断点续传机制**：自动保存每页的 OCR 结果为 JSON，支持异常中断后的快速恢复。

---

## 3. 技术架构
*   **后端**: FastAPI (Python 3.10+)
*   **OCR 引擎**: PaddleOCR (PP-OCRv5 server 系列模型)
*   **PDF 处理**: PyMuPDF (fitz) + ReportLab
*   **前端**: Vue 3 + Element Plus

---

## 4. 关键代码解析

### 4.1 高精度字号匹配逻辑
为了让 PDF 中的文字能够精准覆盖在原图上方，我们实现了基于 PIL 的字号二分搜索算法：
```python
def fit_font_size(text, target_w_px, target_h_px):
    lo, hi = 1, 2000
    best = 12
    while lo <= hi:
        mid = (lo + hi) // 2
        font = get_font(mid)
        tw, th = get_text_size(text, font)
        # 允许一定误差，确保文字能填满识别框
        if tw <= target_w_px + 10 and th <= target_h_px + 10:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return max(6, best // SCALE)
```

### 4.2 多进程页面处理
在 `process_page` 函数中集成 OCR 识别与 PDF 绘制，并支持断点检查：
```python
def process_page(args_tuple):
    # 1. 检查是否已处理过
    if os.path.exists(pdf_path) and os.path.exists(json_path):
        return pdf_path
    # 2. 调用 PaddleOCR predict
    results = ocr.predict(img_path)
    # 3. 实时绘制透明文本层...
```

---

## 5. 快速部署 (Docker)

### 5.1 环境准备
参考 [PaddlePaddle 官方 Docker 安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/fromdocker.html) 获取镜像。

### 5.2 启动容器
```bash
docker run --name paddle_ocr_service \
  -itd \
  -v $PWD:/paddle \
  -p 8038:8038 \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.0 \
  /bin/bash
```

### 5.3 安装依赖与启动
进入容器后执行：
```bash
cd /paddle
# 安装必要依赖
pip install fastapi uvicorn python-multipart pymupdf paddleocr==3.2.0 reportlab opencv-python
# 启动 Web 服务
python main.py
```
访问 `http://localhost:8038` 即可开始使用。

---

## 6. 使用指南
1.  **上传 PDF**：点击页面按钮选择需要处理的文件。
2.  **配置选项**：可勾选“生成带原图背景的可搜索 PDF”（推荐）。
3.  **监控进度**：页面会同步显示当前正处于哪一页的处理阶段。
4.  **下载结果**：处理完成后点击下载按钮获取成品。

---

## 7. 结语
本项目结合了 PaddleOCR 的高识别率与 Python 多进程的高效率，通过 FastAPI 封装为易用的 Web 工具。它不仅解决了 PDF 识别慢的问题，更通过精细的 ReportLab 绘制解决了“看得见、搜不到”的痛点。

希望这篇实战分享能对你有所帮助！如有疑问欢迎在评论区讨论。