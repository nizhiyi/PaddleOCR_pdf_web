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
from reportlab.lib.colors import HexColor,Color
from tqdm import tqdm
import logging
from thread_single import PaddleOCRModelManager
import io


# ① 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ② 构造 current_app（重点）
current_app = type(
    "App",
    (),
    {"logger":logging.getLogger("PaddleOCR")}
)()


# ③ 使用
ocr_manager = PaddleOCRModelManager(current_app=current_app)


# ================== 配置 (对齐 1.py) ==================
BASE_DIR = Path(__file__).parent
TEMP_ROOT    = BASE_DIR / "temp/pdf_pages"
PROGRESS_DIR = BASE_DIR / "temp/progress"
FONT_PATH_CN = BASE_DIR / "fonts/simsun.ttf"
USE_FONT_CN  = "SimSun"
USE_FONT_EN  = "Helvetica"
DEFAULT_DPI  = 350
DEFAULT_WORKERS = 4


# 1.py 中的关键参数
PT_PER_INCH = 72
TEXT_VERTICAL_OFFSET_PX = 10
MASK_PADDING_PX = 4
SCALE = 3


# ================== 命令行参数 ==================
parser = argparse.ArgumentParser()
parser.add_argument("--pdf",      required=True,  help="输入PDF路径")
parser.add_argument("--out",      required=True,  help="输出PDF路径")
parser.add_argument("--task_id",  default="",     help="任务ID，用于进度文件")
parser.add_argument("--inputdpi",  type=int, default=DEFAULT_DPI)
parser.add_argument("--workers",  type=int, default=DEFAULT_WORKERS)
parser.add_argument("--visible",  action="store_true", help="是否显示原图背景 (True=有背景, False=纯白背景)")
parser.add_argument("--keep-temp", action="store_true", help="是否保留临时文件",default=True)
args = parser.parse_args()


input_pdf   = Path(args.pdf).resolve()
output_pdf  = Path(args.out).resolve()
task_id     = args.task_id.strip()
INPUT_PDF_DPI     = args.inputdpi
MAX_WORKERS = max(1, min(args.workers, 8))
KEEP_TEMP   = args.keep_temp
TEXT_VISIBLE = args.visible # 从命令行参数获取背景可见性


if not input_pdf.is_file():
    logging.error(f"❌ 输入文件不存在: {input_pdf}")
    exit(1)


temp_dir = Path(TEMP_ROOT) / task_id if task_id else Path(TEMP_ROOT) / input_pdf.stem
temp_dir.mkdir(parents=True, exist_ok=True)

if INPUT_PDF_DPI > 250:
    OCR_PDF_DPI = 250
else:
    OCR_PDF_DPI = INPUT_PDF_DPI

#if INPUT_PDF_DPI > 350:
#    PDF_DPI = 350
#elif INPUT_PDF_DPI > 50:
#    PDF_DPI = INPUT_PDF_DPI
#else:
#    PDF_DPI = 50
PDF_DPI = INPUT_PDF_DPI
ZOOM_FACTOR = PDF_DPI / OCR_PDF_DPI


# ================== 读取配置（页面选择）====================
config_path = temp_dir / "config.json"
selected_pages = None  # None表示处理所有页面
if config_path.exists():
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        selected_pages = config.get("pages")
        logging.info(f"从配置读取到选择页面: {selected_pages}")
    except Exception as e:
        logging.warning(f"读取配置失败: {e}")


# ================== 进度报告 ==================
processed_pages_list = []  # 记录已处理的页面序号


def save_progress(percent: float, stage: str, done=False):
    if not task_id:
        print(f"[{percent:5.1f}%] {stage}")
        return
    pfile = Path(PROGRESS_DIR) / f"{task_id}.json"
    pfile.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "percent": round(percent, 1), 
        "stage": stage, 
        "done": done,
        "processed_pages": processed_pages_list
    }
    try:
        with open(pfile, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"保存进度失败: {e}")
    logging.info(f"[{percent:5.1f}%] {stage}")


save_progress(0, "启动处理...")


# ================== 字体注册 ==================
if os.path.exists(FONT_PATH_CN):
    try:
        pdfmetrics.registerFont(TTFont(USE_FONT_CN, FONT_PATH_CN))
        logging.info("中文字体注册成功")
    except Exception as e:
        logging.error(f"字体注册失败: {e}，降级使用 Helvetica")
        USE_FONT_CN = "Helvetica"
else:
    USE_FONT_CN = "Helvetica"
    logging.warning("未找到中文字体，降级使用 Helvetica")


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


# ================== 初始化 OCR (保持 130-137 的内容) ==================
save_progress(5, "加载 OCR 模型...")


OFFSET_PX = TEXT_VERTICAL_OFFSET_PX * ZOOM_FACTOR
def process_page(args_tuple):
    """单页 OCR 处理（线程中执行）"""
    img_path, imgh_path, page_num, temp_dir_str = args_tuple
    pdf_path = os.path.join(temp_dir_str, f"ocr_page_{page_num:04d}.pdf")
    json_path = os.path.join(temp_dir_str, f"ocr_page_{page_num:04d}.json")


    if os.path.exists(pdf_path) and os.path.exists(json_path):
        return pdf_path, page_num


    try:
        # 使用全局 OCR 实例（多线程共享，GPU 友好）
        text, raw = ocr_manager.submit_ocr(
            input=img_path,
            json_path=json_path
        )
        # 1.py 的 JSON 读取逻辑
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = data["rec_texts"]
        polys = data["rec_polys"]
        
        # 1.py 的图片读取与缩放逻辑
        orig_img = cv2.imread(imgh_path, cv2.IMREAD_GRAYSCALE)
        if orig_img is None: return None, page_num
        
        img_h, img_w = orig_img.shape[:2]
        pdf_width_pt = img_w * PT_PER_INCH / PDF_DPI
        pdf_height_pt = img_h * PT_PER_INCH / PDF_DPI
        
        c = canvas.Canvas(pdf_path, pagesize=(pdf_width_pt, pdf_height_pt))
        
        if TEXT_VISIBLE:
            # 背景图
            img_pil = Image.fromarray(orig_img)
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
                        # 原图背景模式：变成幽灵文字
                        c.setFillAlpha(0.00)
                        #c.setFillColor(HexColor("#000000"))
                        c.setFillColor(Color(0, 0, 0, alpha=0)) # 完全透明
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
                    # 原图背景模式：变成幽灵文字
                    c.setFillAlpha(0.00)
                    ## c.setFillColor(HexColor("#000000"))
                    c.setFillColor(Color(0, 0, 0, alpha=0)) # 完全透明
                else:
                    c.setFillColor(HexColor("#000000"))
                c.drawCentredString(0, 0, text)
                c.restoreState()


        c.showPage()
        c.save()
        return pdf_path, page_num


    except Exception as e:
        logging.error(f"页 {page_num} 处理出现异常: {e}", exc_info=True)
        # 回退逻辑
        try:
            img = cv2.imread(imgh_path)
            if img is not None:
                h, w = img.shape[:2]
                c = canvas.Canvas(pdf_path, pagesize=(w * 72 / PDF_DPI, h * 72 / PDF_DPI))
                c.drawImage(ImageReader(imgh_path), 0, 0, width=w * 72 / PDF_DPI, height=h * 72 / PDF_DPI)
                c.save()
        except Exception as re:
            logging.error(f"页 {page_num} 回退逻辑也失败了: {re}")
        return pdf_path, page_num


# ================== 主流程 ==================
save_progress(10, "渲染 PDF 页面为图片...")
doc = fitz.open(str(input_pdf))
img_list = []


# 根据选择的页面过滤
total_pages = len(doc)
if selected_pages:
    # 验证页面范围
    valid_pages = [p for p in selected_pages if 1 <= p <= total_pages]
    page_indices = [p - 1 for p in valid_pages]  # 转换为0-based索引
    logging.info(f"将处理选择的页面: {valid_pages}")
else:
    page_indices = list(range(total_pages))

total_rendering = len(page_indices)

page_fordpi = doc[0]
w_inch_pagefordpi = page_fordpi.rect.width / 72
h_inch_pagefordpi = page_fordpi.rect.height / 72
img_fordpi = page_fordpi.get_images(full=True)
if len(img_fordpi) == 0:
    for idx, i in enumerate(page_indices):
        page_num = i + 1  # 1-based页码
        img_path = temp_dir / f"page_{page_num:04d}.png"
        if not img_path.exists():
            pix = doc[i].get_pixmap(dpi=OCR_PDF_DPI)
            pix.save(str(img_path))
        imgh_path = temp_dir / f"pageh_{page_num:04d}.png"
        if not imgh_path.exists():
            pixh = doc[i].get_pixmap(dpi=PDF_DPI, colorspace=fitz.csGRAY, alpha=False)
            pixh.pil_save(str(imgh_path), compress_level=9, optimize=True)
        img_list.append((str(img_path), str(imgh_path), page_num, str(temp_dir)))
        render_percent = 10 + (idx + 1) / total_rendering * 15
        save_progress(render_percent, f"渲染页面 {page_num}/{total_pages}")
else:
    max_area = 0
    for imgd in img_fordpi:
        xref = imgd[0]
        area = imgd[2] * imgd[3]
        if area > max_area:
            max_area = area
            largest_image_xref = xref
    w_inch_fordpi = page_fordpi.get_image_rects(largest_image_xref)[0].width / 72
    h_inch_fordpi = page_fordpi.get_image_rects(largest_image_xref)[0].height / 72
    if w_inch_fordpi / w_inch_pagefordpi < 0.99 or h_inch_fordpi / h_inch_pagefordpi < 0.99:
        for idx, i in enumerate(page_indices):
            page_num = i + 1  # 1-based页码
            img_path = temp_dir / f"page_{page_num:04d}.png"
            if not img_path.exists():
                pix = doc[i].get_pixmap(dpi=OCR_PDF_DPI)
                pix.save(str(img_path))
            imgh_path = temp_dir / f"pageh_{page_num:04d}.png"
            if not imgh_path.exists():
                pixh = doc[i].get_pixmap(dpi=PDF_DPI, colorspace=fitz.csGRAY, alpha=False)
                pixh.pil_save(str(imgh_path), compress_level=9, optimize=True)
            img_list.append((str(img_path), str(imgh_path), page_num, str(temp_dir)))
            render_percent = 10 + (idx + 1) / total_rendering * 15
            save_progress(render_percent, f"渲染页面 {page_num}/{total_pages}")  
    else:
        for idx, i in enumerate(page_indices):
            page_num = i + 1  # 1-based页码
            img_path = temp_dir / f"page_{page_num:04d}.png"
            if not img_path.exists():
                pix = doc[i].get_pixmap(dpi=OCR_PDF_DPI)
                pix.save(str(img_path))
            img_origin = doc[i].get_images(full=True)
            if len(img_origin) == 0:
                imgh_path = temp_dir / f"pageh_{page_num:04d}.png"
                pixh = doc[i].get_pixmap(dpi=PDF_DPI, colorspace=fitz.csGRAY, alpha=False)
                pixh.pil_save(str(imgh_path), compress_level=9, optimize=True)
            else:
                max_area = 0
                for imgd in img_origin:
                    xref = imgd[0]
                    area = imgd[2] * imgd[3]
                    if area > max_area:
                        max_area = area
                        largest_image_xref = xref
                page_origin = doc.extract_image(largest_image_xref)
                image_bytes = page_origin["image"]
                image_ext = page_origin["ext"]
                imgh_path = temp_dir / f"pageh_{page_num:04d}.{image_ext}"
                rot = doc[i].rotation
                if rot != 0 and rot % 90 == 0:
                    imgrot = Image.open(io.BytesIO(image_bytes))
                    if rot == 90:
                        imgrot = imgrot.rotate(270, expand=True)   # 顺时针90°
                    elif rot == 180:
                        imgrot = imgrot.rotate(180, expand=True)
                    elif rot == 270:
                        imgrot = imgrot.rotate(90, expand=True)    # 逆时针90°
                    imgrot.save(str(imgh_path))
                else:
                    Path(imgh_path).write_bytes(image_bytes)
            img_list.append((str(img_path), str(imgh_path), page_num, str(temp_dir)))
            render_percent = 10 + (idx + 1) / total_rendering * 15
            save_progress(render_percent, f"渲染页面 {page_num}/{total_pages}")
doc.close()


save_progress(25, f"启动 {len(img_list)} 页的 OCR 任务...")
tasks = img_list
page_pdfs = [None] * len(tasks)
page_nums = [None] * len(tasks)


count = 0
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_idx = {executor.submit(process_page, t): i for i, t in enumerate(tasks)}
    for future in tqdm(as_completed(future_to_idx), total=len(tasks), desc="OCR 处理"):
        idx = future_to_idx[future]
        try:
            pdf_path, page_num = future.result()
            page_pdfs[idx] = pdf_path
            page_nums[idx] = page_num
            # 记录已处理的页面
            if page_num and page_num not in processed_pages_list:
                processed_pages_list.append(page_num)
                processed_pages_list.sort()
        except Exception as e:
            logging.error(f"进程执行任务时出错 (任务索引 {idx}): {e}", exc_info=True)
        count += 1
        ocr_percent = 25 + (count / len(tasks)) * 65
        save_progress(ocr_percent, f"OCR 处理中 {count}/{len(tasks)}")


save_progress(90, "正在合并 PDF...")
final_doc = fitz.open()
for p in page_pdfs:
    if p and os.path.exists(p):
        src = fitz.open(p)
        final_doc.insert_pdf(src)
        src.close()


final_doc.save(str(output_pdf), garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, clean=True, use_objstms=1)
final_doc.close()


if not KEEP_TEMP:
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


save_progress(100, "处理完成", done=True)
logging.info(f"✅ 完成！输出文件：{output_pdf}")