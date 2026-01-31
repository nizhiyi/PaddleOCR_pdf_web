import os
import math
import argparse
import json
import cv2
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.colors import HexColor
from tqdm import tqdm
from paddleocr import PaddleOCR

# ================== 日志配置 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ocr_process.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ================== 配置 (对齐 1.py) ==================
TEMP_ROOT    = "./pdf_pages"
PROGRESS_DIR = "./progress"
FONT_PATH_CN = "./fonts/simsun.ttf"
USE_FONT_CN  = "SimSun"
USE_FONT_EN  = "Helvetica"
DEFAULT_DPI  = 150 # 对齐 1.py 的 IMG_DPI
DEFAULT_WORKERS = 4

# 1.py 中的关键参数
IMG_DPI = 150
PDF_DPI = 150
PT_PER_INCH = 72
ZOOM_FACTOR = 2.0
TEXT_VERTICAL_OFFSET_PX = 10
MASK_PADDING_PX = 4
SCALE = 3

# ================== 命令行参数 ==================
parser = argparse.ArgumentParser()
parser.add_argument("--pdf",      required=True,  help="输入PDF路径")
parser.add_argument("--out",      required=True,  help="输出PDF路径")
parser.add_argument("--task_id",  default="",     help="任务ID，用于进度文件")
parser.add_argument("--dpi",      type=int, default=DEFAULT_DPI)
parser.add_argument("--workers",  type=int, default=DEFAULT_WORKERS)
parser.add_argument("--visible",  action="store_true", help="是否显示原图背景 (True=有背景, False=纯白背景)")
parser.add_argument("--keep-temp", action="store_true")
args = parser.parse_args()

input_pdf   = Path(args.pdf).resolve()
output_pdf  = Path(args.out).resolve()
task_id     = args.task_id.strip()
DPI         = args.dpi
MAX_WORKERS = max(1, min(args.workers, 8))
KEEP_TEMP   = args.keep_temp
TEXT_VISIBLE = args.visible # 从命令行参数获取背景可见性

if not input_pdf.is_file():
    logger.error(f"❌ 输入文件不存在: {input_pdf}")
    exit(1)

temp_dir = Path(TEMP_ROOT) / input_pdf.stem
temp_dir.mkdir(parents=True, exist_ok=True)

# ================== 进度报告 ==================
def save_progress(percent: float, stage: str, done=False):
    if not task_id:
        print(f"[{percent:5.1f}%] {stage}")
        return
    pfile = Path(PROGRESS_DIR) / f"{task_id}.json"
    pfile.parent.mkdir(parents=True, exist_ok=True)
    data = {"percent": round(percent, 1), "stage": stage, "done": done}
    try:
        with open(pfile, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"保存进度失败: {e}")
    logger.info(f"[{percent:5.1f}%] {stage}")

save_progress(0, "启动处理...")

# ================== 字体注册 ==================
if os.path.exists(FONT_PATH_CN):
    try:
        pdfmetrics.registerFont(TTFont(USE_FONT_CN, FONT_PATH_CN))
        logger.info("中文字体注册成功")
    except Exception as e:
        logger.error(f"字体注册失败: {e}，降级使用 Helvetica")
        USE_FONT_CN = "Helvetica"
else:
    USE_FONT_CN = "Helvetica"
    logger.warning("未找到中文字体，降级使用 Helvetica")

def get_font_name(ch):
    return USE_FONT_CN if '\u4e00' <= ch <= '\u9fff' else USE_FONT_EN

# ================== 1.py 字体适配逻辑 ==================
font_cache = {}
def get_font(size_pt: int):
    size_pt = max(1, int(size_pt))
    if size_pt not in font_cache:
        font_cache[size_pt] = ImageFont.truetype(FONT_PATH_CN, size_pt)
    return font_cache[size_pt]

def get_text_size(text, font):
    dummy = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def fit_font_size_for_char(char: str, target_w_px: float, target_h_px: float) -> int:
    if not char.strip(): return 12
    lo, hi = 1, 2000
    best = 12
    target_w = target_w_px * PT_PER_INCH / PDF_DPI * SCALE
    target_h = target_h_px * PT_PER_INCH / PDF_DPI * SCALE
    for _ in range(25):
        mid = (lo + hi) // 2
        font = get_font(mid)
        tw, th = get_text_size(char, font)
        if tw <= target_w + 10 and th <= target_h + 10:
            best, lo = mid, mid + 1
        else:
            hi = mid
    return max(6, best // SCALE)

def fit_font_size(text: str, target_w_px: float, target_h_px: float) -> int:
    if not text.strip(): return 12
    lo, hi = 1, 2000
    best = 12
    target_w = target_w_px * PT_PER_INCH / PDF_DPI * SCALE
    target_h = target_h_px * PT_PER_INCH / PDF_DPI * SCALE
    for _ in range(25):
        mid = (lo + hi) // 2
        font = get_font(mid)
        tw, th = get_text_size(text, font)
        if tw <= target_w + 10 and th <= target_h + 10:
            best, lo = mid, mid + 1
        else:
            hi = mid
    return max(6, best // SCALE)

def px_to_pt(x_px, y_px, pdf_height_pt):
    return x_px * PT_PER_INCH / PDF_DPI, pdf_height_pt - y_px * PT_PER_INCH / PDF_DPI

# ================== GPU 检测 ==================
def get_device():
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            logger.info("检测到可用 GPU，多线程模式下启用 GPU 加速")
            return "gpu:0"
    except Exception as e:
        logger.warning(f"GPU 检测失败: {e}")
    logger.info("使用 CPU 进行 OCR")
    return "cpu"

# ================== 初始化 OCR (保持 130-137 的内容) ==================
save_progress(5, "加载 OCR 模型...")

def init_ocr():
    """初始化 OCR，支持 GPU 加速（多线程共享）"""
    device = get_device()
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device
    )
    logger.info(f"OCR 模型使用 {device} 初始化成功")
    return ocr

OFFSET_PX = TEXT_VERTICAL_OFFSET_PX * ZOOM_FACTOR

# ================== OCR 单例 (多线程共享) ==================
OCR_INSTANCE = None

def get_ocr_instance():
    """获取全局 OCR 实例，多线程共享（GPU 友好）"""
    global OCR_INSTANCE
    if OCR_INSTANCE is None:
        OCR_INSTANCE = init_ocr()
    return OCR_INSTANCE

def process_page(args_tuple):
    """单页 OCR 处理（线程中执行）"""
    img_path, page_num, temp_dir_str = args_tuple
    pdf_path = os.path.join(temp_dir_str, f"ocr_page_{page_num:04d}.pdf")
    json_path = os.path.join(temp_dir_str, f"ocr_page_{page_num:04d}.json")

    if os.path.exists(pdf_path) and os.path.exists(json_path):
        return pdf_path

    try:
        # 使用全局 OCR 实例（多线程共享，GPU 友好）
        ocr = get_ocr_instance()
        
        # 保持 ocr.predict(img_path) 不变
        results = ocr.predict(img_path)
        
        # 1.py 的 JSON 保存逻辑 (results 是列表，内部元素有 save_to_json)
        try:
            for res in results:
                res.save_to_json(json_path)
        except Exception as je:
            logger.error(f"保存 JSON 失败 (页 {page_num}): {je}")
        
        # 1.py 的 JSON 读取逻辑
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = data["rec_texts"]
        polys = data["rec_polys"]
        
        # 1.py 的图片读取与缩放逻辑
        orig_img = cv2.imread(img_path)
        if orig_img is None: return None
        
        img_h, img_w = orig_img.shape[:2]
        img_w_zoom = int(img_w * ZOOM_FACTOR)
        img_h_zoom = int(img_h * ZOOM_FACTOR)
        pdf_width_pt = img_w_zoom * PT_PER_INCH / PDF_DPI
        pdf_height_pt = img_h_zoom * PT_PER_INCH / PDF_DPI
        
        c = canvas.Canvas(pdf_path, pagesize=(pdf_width_pt, pdf_height_pt))
        
        if TEXT_VISIBLE:
            # 背景图
            img_zoom = cv2.resize(orig_img, (img_w_zoom, img_h_zoom), interpolation=cv2.INTER_LANCZOS4)
            img_pil = Image.fromarray(cv2.cvtColor(img_zoom, cv2.COLOR_BGR2RGB))
            c.drawImage(ImageReader(img_pil), 0, 0, width=pdf_width_pt, height=pdf_height_pt)
        else:
            # 白色背景
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, pdf_width_pt, pdf_height_pt, fill=1, stroke=0)

        # 1.py 的文本层绘制逻辑
        for text, poly in zip(texts, polys):
            if not text.strip(): continue
            box = np.array(poly, dtype=np.float32).reshape(4, 2) * ZOOM_FACTOR
            x0, y0 = box.min(axis=0)
            x1, y1 = box.max(axis=0)
            w_box, h_box = x1 - x0, y1 - y0

            angle_deg = np.degrees(np.arctan2(box[1][1] - box[0][1], box[1][0] - box[0][0]))
            vertical = h_box > 2.5 * w_box and h_box > 120

            if vertical:
                chars = [ch for ch in text if ch.strip()]
                if not chars: continue
                per_char_w_px = w_box * 0.90
                per_char_h_px = h_box * 0.80 / len(chars)
                font_size_pt = min([fit_font_size_for_char(ch, per_char_w_px, per_char_h_px) for ch in chars] or [12])
                gap_pt = max(2, font_size_pt * 0.30)
                total_h_pt = len(chars) * font_size_pt + (len(chars) - 1) * gap_pt
                total_h_px = total_h_pt * PDF_DPI / PT_PER_INCH
                y_start_px = y0 + (h_box - total_h_px) / 2 + OFFSET_PX
                cx_px = (x0 + x1) / 2
                cur_y_px = y_start_px

                for ch in text:
                    if not ch.strip(): continue
                    baseline_offset_px = font_size_pt * PDF_DPI / PT_PER_INCH / 2
                    x_pt, y_pt = px_to_pt(cx_px, cur_y_px + baseline_offset_px, pdf_height_pt)
                    c.saveState()
                    c.translate(x_pt, y_pt)
                    c.rotate(-angle_deg)
                    c.setFont(USE_FONT_CN, font_size_pt)
                    if TEXT_VISIBLE:
                        c.setFillAlpha(0.0) 
                    else:
                        c.setFillColor(HexColor("#000000"))
                    c.drawCentredString(0, 0, ch)
                    c.restoreState()
                    cur_y_px += (font_size_pt + gap_pt) * PDF_DPI / PT_PER_INCH
            else:
                font_size_pt = fit_font_size(text, w_box, h_box)
                cx_px = (x0 + x1) / 2
                cy_px = (y0 + y1) / 2 + OFFSET_PX
                x_pt, y_pt = px_to_pt(cx_px, cy_px, pdf_height_pt)
                c.saveState()
                c.translate(x_pt, y_pt)
                c.rotate(-angle_deg)
                c.setFont(USE_FONT_CN, font_size_pt)
                if TEXT_VISIBLE:
                    c.setFillAlpha(0.0) 
                else:
                    c.setFillColor(HexColor("#000000"))
                c.drawCentredString(0, 0, text)
                c.restoreState()

        c.showPage()
        c.save()
        return pdf_path

    except Exception as e:
        logger.error(f"页 {page_num} 处理出现异常: {e}", exc_info=True)
        # 回退逻辑
        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                c = canvas.Canvas(pdf_path, pagesize=(w * 72 / DPI, h * 72 / DPI))
                c.drawImage(ImageReader(img_path), 0, 0, width=w * 72 / DPI, height=h * 72 / DPI)
                c.save()
        except Exception as re:
            logger.error(f"页 {page_num} 回退逻辑也失败了: {re}")
        return pdf_path

# ================== 主流程 ==================
save_progress(10, "渲染 PDF 页面为图片...")
doc = fitz.open(str(input_pdf))
img_list = []
total_rendering = len(doc)
for i in tqdm(range(total_rendering), desc="页面渲染"):
    img_path = temp_dir / f"page_{i+1:04d}.png"
    if not img_path.exists():
        pix = doc[i].get_pixmap(dpi=DPI)
        pix.save(str(img_path))
    img_list.append(str(img_path))
    render_percent = 10 + (i + 1) / total_rendering * 15
    save_progress(render_percent, f"渲染页面 {i+1}/{total_rendering}")
doc.close()

save_progress(25, f"启动 {len(img_list)} 页的 OCR 任务...")
tasks = [(img, i+1, str(temp_dir)) for i, img in enumerate(img_list)]
page_pdfs = [None] * len(tasks)

count = 0
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_idx = {executor.submit(process_page, t): i for i, t in enumerate(tasks)}
    for future in tqdm(as_completed(future_to_idx), total=len(tasks), desc="OCR 处理"):
        idx = future_to_idx[future]
        try:
            page_pdfs[idx] = future.result()
        except Exception as e:
            logger.error(f"进程执行任务时出错 (任务索引 {idx}): {e}", exc_info=True)
        count += 1
        ocr_percent = 25 + (count / len(tasks)) * 65
        save_progress(ocr_percent, f"OCR 处理中 {count}/{len(tasks)}")

save_progress(90, "正在合并 PDF...")
final_doc = fitz.open()
for p in page_pdfs:
    if p:
        src = fitz.open(p)
        final_doc.insert_pdf(src)
        src.close()

final_doc.save(str(output_pdf), garbage=4, deflate=True, clean=True)
final_doc.close()

#if not KEEP_TEMP:
    #import shutil
    #shutil.rmtree(temp_dir, ignore_errors=True)

save_progress(100, "处理完成", done=True)
logger.info(f"✅ 完成！输出文件：{output_pdf}")
