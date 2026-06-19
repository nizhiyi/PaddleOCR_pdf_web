from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
import os
import uuid
import subprocess
import json
import time
import shutil
import fitz


BASE = "."
PDF_DIR = os.path.abspath("temp/input_pdfs")
OUT_DIR = os.path.abspath("temp/output_pdfs")
PROGRESS_DIR = os.path.abspath("temp/progress")
PAGES_DIR = os.path.abspath("temp/pdf_pages")
UPLOAD_DIR = os.path.abspath("temp/uploads")


for d in [PDF_DIR, OUT_DIR, PROGRESS_DIR, PAGES_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)


app = FastAPI(title="PaddleOCR PDF 可搜索化服务")


# ================== 前端单页 ==================
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>PaddleOCR PDF - 制作可搜索PDF</title>


  <!-- Element Plus -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/element-plus@2.9.7/dist/index.css" />
  <script src="https://cdn.jsdelivr.net/npm/vue@3/dist/vue.global.prod.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/element-plus@2.9.7/dist/index.full.min.js"></script>
  <!-- Element Plus Icons -->
  <script src="https://cdn.jsdelivr.net/npm/@element-plus/icons-vue@2.3.2"></script>

  <!-- PDF.js -->
  <script src="https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js"></script>


  <style>
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background:#f5f7fa;
      margin:0;
      padding:20px;
    }
    #app {
      max-width:1400px;
      margin:0 auto;
      background:white;
      padding:40px;
      border-radius:16px;
      box-shadow:0 4px 20px rgba(0,0,0,0.1);
    }
    .header { text-align:center; margin-bottom:40px; }
    .upload-area { text-align:center; padding:40px 0; }
    .progress-box { margin:32px 0; }
    .result-box { margin-top:40px; text-align:center; }
    
    /* 页面选择样式 */
    .page-select-container {
      margin: 20px 0;
      max-height: 300px;
      overflow-y: auto;
      border: 1px solid #dcdfe6;
      border-radius: 8px;
      padding: 15px;
    }
    .page-select-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 15px;
      padding-bottom: 10px;
      border-bottom: 1px solid #ebeef5;
    }
    .page-list {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .page-item {
      position: relative;
      width: 100px;
      cursor: pointer;
      border: 2px solid transparent;
      border-radius: 6px;
      overflow: hidden;
      transition: all 0.3s;
    }
    .page-item:hover {
      border-color: #409eff;
    }
    .page-item.selected {
      border-color: #409eff;
      box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
    }
    .page-item img {
      width: 100%;
      height: 140px;
      object-fit: cover;
      display: block;
    }
    .page-item .page-num {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(0,0,0,0.6);
      color: white;
      text-align: center;
      padding: 4px;
      font-size: 12px;
    }
    .page-item .check-icon {
      position: absolute;
      top: 5px;
      right: 5px;
      background: #409eff;
      color: white;
      border-radius: 50%;
      width: 20px;
      height: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
    }
    
    /* 实时预览样式 */
    .preview-container {
      margin-top: 30px;
    }
    .preview-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 15px;
    }
    .preview-layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      height: 600px;
    }
    .preview-panel {
      border: 1px solid #dcdfe6;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .preview-panel-header {
      background: #f5f7fa;
      padding: 10px 15px;
      font-weight: bold;
      border-bottom: 1px solid #dcdfe6;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .preview-content {
      flex: 1;
      overflow: auto;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f0f0f0;
    }
    .preview-content img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    #pdf-canvas {
      max-width: 100%;
      max-height: 100%;
    }
    
    /* 页面导航 */
    .page-nav {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .page-nav input {
      width: 60px;
      text-align: center;
    }
  </style>
</head>


<body>
<div id="app">
  <div class="header">
    <h2>PaddleOCR PDF</h2>
    <p style="color:#666;">上传 PDF → OCR 识别 → 生成可搜索 PDF</p>
  </div>


  <!-- 上传区 -->
  <div class="upload-area" v-if="step === 'upload'">
    <el-upload
      :auto-upload="false"
      :limit="1"
      accept=".pdf"
      :on-change="handleFileChange"
    >
      <el-button type="primary" size="large">
        <el-icon style="margin-right:6px"><Upload /></el-icon>
        选择 PDF 文件
      </el-button>
    </el-upload>
  </div>


  <!-- 页面选择和配置区 -->
  <div v-if="step === 'select' && pages.length > 0">
    <el-card shadow="hover">
      <template #header>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span>选择要转换的页面</span>
          <el-tag type="info">共 {{ pages.length }} 页</el-tag>
        </div>
      </template>
      
      <!-- 页面选择 -->
      <div class="page-select-container">
        <div class="page-select-header">
          <div style="display: flex; align-items: center; gap: 20px;">
            <el-checkbox v-model="selectAll" @change="handleSelectAll">全选</el-checkbox>
            <span style="color: #666; font-size: 14px;">已选择 {{ selectedPages.length }} 页</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <el-checkbox v-model="isVisible" size="small" border>
              显示原图背景
            </el-checkbox>
            <el-tooltip content="勾选后生成的PDF保留原图背景，文字几乎透明但可搜索复制；不勾选则为纯白背景黑字" placement="top">
              <el-icon style="color: #909399; cursor: help;"><QuestionFilled /></el-icon>
            </el-tooltip>
          </div>
        </div>
        <div class="page-list">
          <div 
            v-for="page in pages" 
            :key="page.num"
            class="page-item"
            :class="{ selected: selectedPages.includes(page.num) }"
            @click="togglePage(page.num)"
          >
            <img :src="page.thumb" :alt="'Page ' + page.num">
            <div class="page-num">第 {{ page.num }} 页</div>
            <div v-if="selectedPages.includes(page.num)" class="check-icon">
              <el-icon><Check /></el-icon>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 操作按钮 -->
      <div style="text-align: center; margin-top: 20px;">
        <el-button size="large" @click="reset" style="margin-right: 10px;">重新上传</el-button>
        <el-button 
          type="primary" 
          size="large" 
          @click="startOcr"
          :disabled="selectedPages.length === 0"
        >
          开始 OCR 处理
        </el-button>
      </div>
    </el-card>
  </div>


  <!-- 进度和预览区 -->
  <div v-if="step === 'processing' || step === 'done'">
    <el-card shadow="hover" class="progress-box">
      <template #header>
        <div style="display:flex; justify-content:space-between;">
          <span>处理进度</span>
          <el-tag v-if="done" type="success">已完成</el-tag>
        </div>
      </template>


      <p><strong>当前阶段：</strong>{{ stage || '处理中...' }}</p>


      <el-progress
        :percentage="percent"
        :stroke-width="20"
        text-inside
        :status="percent === 100 ? 'success' : 'active'"
      />
    </el-card>


    <!-- 实时预览 -->
    <div v-if="processedPages.length > 0" class="preview-container">
      <el-card shadow="hover">
        <div class="preview-header">
          <div style="display: flex; align-items: center; gap: 15px;">
            <span style="font-weight: bold;">实时预览</span>
            <el-checkbox v-model="autoFollow" size="small" border>
              自动跟随最新页
            </el-checkbox>
          </div>
          <div class="page-nav">
            <el-button size="mini" @click="prevPage" :disabled="currentPreviewPage <= 1"> - </el-button>
            <span><el-input v-model.number="currentPreviewPageInput" size="mini" @change="jumpToPage" /> / {{ processedPages.length }}</span>
            <el-button size="mini" @click="nextPage" :disabled="currentPreviewPage >= processedPages.length"> + </el-button>
          </div>
        </div>
        
        <div class="preview-layout">
          <!-- 左侧：原图 -->
          <div class="preview-panel">
            <div class="preview-panel-header">
              <span>原始图片</span>
              <el-tag size="small" type="info">原图</el-tag>
            </div>
            <div class="preview-content">
              <img v-if="currentOriginalImage" :src="currentOriginalImage" alt="原始图片">
              <el-empty v-else description="加载中..." />
            </div>
          </div>
          
          <!-- 右侧：PDF预览 (使用iframe) -->
          <div class="preview-panel">
            <div class="preview-panel-header">
              <span>OCR 结果 (可复制文字)</span>
              <el-tag size="small" type="success">可搜索PDF</el-tag>
            </div>
            <div class="preview-content" style="padding: 0;">
              <iframe 
                v-if="currentPdfUrl" 
                :src="currentPdfUrl + '#toolbar=1&navpanes=0'" 
                width="100%" 
                height="100%" 
                style="border: none;"
              ></iframe>
              <el-empty v-else description="等待处理..." />
            </div>
          </div>
        </div>
      </el-card>
    </div>


    <!-- 处理完成弹窗 -->
    <el-dialog
      v-model="showCompleteDialog"
      title="🎉 处理完成"
      width="400px"
      :close-on-click-modal="false"
      :show-close="false"
      center
    >
      <div style="text-align: center; padding: 20px 0;">
        <el-icon :size="60" color="#67c23a" style="margin-bottom: 15px;"><CircleCheck /></el-icon>
        <p style="font-size: 16px; color: #606266; margin: 0;">
          已生成 <strong style="color: #409eff;">{{ processedPages.length }}</strong> 页可搜索 PDF
        </p>
      </div>
      <template #footer>
        <div style="display: flex; justify-content: center; gap: 15px;">
          <el-button type="primary" size="large" @click="download">
            <el-icon style="margin-right: 5px;"><Download /></el-icon>
            下载 PDF
          </el-button>
          <el-button size="large" @click="closeCompleteDialog">
            再处理一个
          </el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</div>


<script>
const { createApp, ref, computed, watch, nextTick } = Vue


const app = createApp({
  setup() {
    // 步骤：upload -> select -> processing -> done
    const step = ref('upload')
    const taskId = ref('')
    const pages = ref([])
    const indpi = ref(0)
    const selectedPages = ref([])
    const selectAll = ref(true)
    const isVisible = ref(true)  // 默认不勾选生成带原图背景
    
    // 进度
    const percent = ref(0)
    const stage = ref('')
    const done = ref(false)
    const showCompleteDialog = ref(false)
    
    // 预览
    const processedPages = ref([])
    const currentPreviewPage = ref(1)
    const currentPreviewPageInput = ref(1)
    const autoFollow = ref(true)  // 默认开启自动跟随
    
    const currentOriginalImage = computed(() => {
      if (processedPages.value.length === 0) return ''
      const page = processedPages.value[currentPreviewPage.value - 1]
      return page ? `/preview/img/${taskId.value}/${page}` : ''
    })
    const currentPdfUrl = computed(() => {
      if (processedPages.value.length === 0) return ''
      const page = processedPages.value[currentPreviewPage.value - 1]
      return page ? `/preview/pdf/${taskId.value}/${page}` : ''
    })
    
    // 同步input和实际页码
    watch(currentPreviewPage, (val) => {
      currentPreviewPageInput.value = val
    })
    
    // 自动跟随最新处理的页面
    watch(processedPages, (pages) => {
      if (autoFollow.value && pages.length > 0) {
        // 自动跳转到最新处理的页面
        currentPreviewPage.value = pages.length
      }
    }, { immediate: false })


    const handleFileChange = async (fileItem) => {
      if (!fileItem?.raw) return
      
      const fd = new FormData()
      fd.append('file', fileItem.raw)
      
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: fd
      })
      const data = await res.json()
      
      if (data.task_id) {
        taskId.value = data.task_id
        pages.value = data.pages.map(p => ({
          num: p.num,
          thumb: `/preview/img/${data.task_id}/${p.num}`
        }))
        indpi.value = data.inputdpi
        selectedPages.value = data.pages.map(p => p.num)
        selectAll.value = true
        step.value = 'select'
      }
    }
    
    const handleSelectAll = (val) => {
      if (val) {
        selectedPages.value = pages.value.map(p => p.num)
      } else {
        selectedPages.value = []
      }
    }
    
    const togglePage = (num) => {
      const idx = selectedPages.value.indexOf(num)
      if (idx > -1) {
        selectedPages.value.splice(idx, 1)
      } else {
        selectedPages.value.push(num)
      }
      selectAll.value = selectedPages.value.length === pages.value.length
    }


    const startOcr = async () => {
      step.value = 'processing'
      
      // 发送OCR请求
      const res = await fetch('/api/ocr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_id: taskId.value,
          pages: selectedPages.value,
          inputdpi: indpi.value,
          visible: isVisible.value
        })
      })
      
      listenSSE()
    }
    
    const listenSSE = () => {
      const es = new EventSource(`/api/sse/${taskId.value}`)
      es.onmessage = e => {
        const d = JSON.parse(e.data)
        percent.value = d.percent || 0
        stage.value = d.stage || ''
        
        // 更新已处理的页面列表
        if (d.processed_pages) {
          processedPages.value = d.processed_pages
          // 如果是第一页处理完成，开始预览
          if (processedPages.value.length === 1) {
            currentPreviewPage.value = 1
          }
        }
        
        if (d.done) {
          done.value = true
          es.close()
          step.value = 'done'
          // 显示完成弹窗
          setTimeout(() => {
            showCompleteDialog.value = true
          }, 500)
        }
      }
    }
    
    const prevPage = () => {
      if (currentPreviewPage.value > 1) {
        currentPreviewPage.value--
        autoFollow.value = false  // 手动切换时关闭自动跟随
      }
    }
    
    const nextPage = () => {
      if (currentPreviewPage.value < processedPages.value.length) {
        currentPreviewPage.value++
        autoFollow.value = false  // 手动切换时关闭自动跟随
      }
    }
    
    const jumpToPage = () => {
      let page = parseInt(currentPreviewPageInput.value)
      if (isNaN(page)) page = 1
      if (page < 1) page = 1
      if (page > processedPages.value.length) page = processedPages.value.length
      currentPreviewPage.value = page
      // 用户手动跳转时，关闭自动跟随
      autoFollow.value = false
    }


    const download = () => {
      if (taskId.value && done.value) {
        const a = document.createElement('a')
        a.href = `/download/${taskId.value}`
        a.download = `ocr_result_${taskId.value}.pdf`
        a.click()
      }
    }


    const reset = () => {
      step.value = 'upload'
      taskId.value = ''
      pages.value = []
      indpi.value = 0
      selectedPages.value = []
      selectAll.value = true
      isVisible.value = true
      percent.value = 0
      stage.value = ''
      done.value = false
      processedPages.value = []
      currentPreviewPage.value = 1
      currentPreviewPageInput.value = 1
      showCompleteDialog.value = false
    }


    const closeCompleteDialog = () => {
      showCompleteDialog.value = false
      setTimeout(() => {
        reset()
      }, 300)
    }


    return { 
      step, taskId, pages, selectedPages, selectAll, isVisible,
      percent, stage, done, showCompleteDialog, processedPages, currentPreviewPage, currentPreviewPageInput,
      currentOriginalImage, currentPdfUrl, autoFollow,
      handleFileChange, handleSelectAll, togglePage, startOcr,
      prevPage, nextPage, jumpToPage, download, reset, closeCompleteDialog
    }
  }
})


/* 注册 Element Plus */
app.use(ElementPlus)


/* 注册 Icons */
Object.entries(ElementPlusIconsVue).forEach(([name, comp]) => {
  app.component(name, comp)
})


app.mount('#app')
</script>


</body>
</html>
    """


@app.post("/api/upload")
async def upload(file: UploadFile):
    """上传PDF，渲染页面缩略图"""
    tid = str(uuid.uuid4())
    pdf_path = os.path.join(PDF_DIR, f"{tid}.pdf")
    task_pages_dir = os.path.join(PAGES_DIR, tid)
    
    os.makedirs(task_pages_dir, exist_ok=True)
    
    try:
        # 保存上传的PDF
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}")
    
    # 使用 PyMuPDF 渲染页面缩略图
    try:
        doc = fitz.open(pdf_path)
        pages = []
        
        page_fordpi = doc[0]
        w_inch_pagefordpi = page_fordpi.rect.width / 72
        h_inch_pagefordpi = page_fordpi.rect.height / 72
        img_fordpi = page_fordpi.get_images(full=True)
        if len(img_fordpi) == 0:
            INPUT_PDF_DPI = 350
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
            pix_w_fordpi = doc.extract_image(largest_image_xref)["width"]
            if (w_inch_fordpi * h_inch_fordpi) / (w_inch_pagefordpi * h_inch_pagefordpi) < 0.30:
                INPUT_PDF_DPI = 350
            elif w_inch_fordpi / w_inch_pagefordpi < 0.99 or h_inch_fordpi / h_inch_pagefordpi < 0.99:
                INPUT_PDF_DPI = min(round(pix_w_fordpi / w_inch_fordpi), 350)
            else:
                INPUT_PDF_DPI = round(pix_w_fordpi / w_inch_fordpi)
            
        if INPUT_PDF_DPI > 250:
            OCR_PDF_DPI = 250
        else:
            OCR_PDF_DPI = INPUT_PDF_DPI

        for i in range(len(doc)):
            page_num = i + 1
            img_path = os.path.join(task_pages_dir, f"page_{page_num:04d}.png")
            
            # 渲染低分辨率缩略图用于预览
            page = doc[i]
            pix = page.get_pixmap(dpi=OCR_PDF_DPI)
            pix.save(img_path)
            
            pages.append({"num": page_num, "img": f"page_{page_num:04d}.png"})
        
        doc.close()
        
        return {
            "task_id": tid,
            "pages": pages,
            "inputdpi": INPUT_PDF_DPI
        }
    except Exception as e:
        # 清理
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if os.path.exists(task_pages_dir):
            shutil.rmtree(task_pages_dir)
        raise HTTPException(status_code=500, detail=f"PDF渲染失败: {e}")



@app.post("/api/ocr")
async def ocr(data: dict):
    """启动OCR处理"""
    tid = data.get("task_id")
    pages = data.get("pages", [])
    INPUT_PDF_DPI = data.get("inputdpi")
    visible = data.get("visible", True)
    
    if not tid or not pages or not INPUT_PDF_DPI:
        raise HTTPException(status_code=400, detail="缺少task_id或pages或INPUT_PDF_DPI")
    
    pdf_path = os.path.join(PDF_DIR, f"{tid}.pdf")
    out_path = os.path.join(OUT_DIR, f"{tid}.pdf")
    task_pages_dir = os.path.join(PAGES_DIR, tid)
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF文件不存在")
    
    # 保存配置
    config = {
        "pages": pages,
        "visible": visible,
        "total_pages": len(pages),
        "processed": []
    }
    config_path = os.path.join(task_pages_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)
    
    # 启动OCR进程
    cmd = [
        "python",
        "pdf_ocr_select.py",
        "--pdf", pdf_path,
        "--out", out_path,
        "--task_id", tid,
        "--workers", "4",
        "--inputdpi", str(INPUT_PDF_DPI)
    ]
    if visible:
        cmd.append("--visible")
    
    print(f"启动 OCR 进程: {' '.join(cmd)}")
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动处理进程失败: {e}")
    
    return {"task_id": tid}



@app.get("/preview/img/{tid}/{page_num}")
async def preview_image(tid: str, page_num: int):
    """获取页面原图预览"""
    img_path = os.path.join(PAGES_DIR, tid, f"page_{page_num:04d}.png")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(img_path, media_type="image/png")



@app.get("/preview/pdf/{tid}/{page_num}")
async def preview_pdf(tid: str, page_num: int):
    """获取单页PDF预览"""
    pdf_path = os.path.join(PAGES_DIR, tid, f"ocr_page_{page_num:04d}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF页面不存在")
    return FileResponse(pdf_path, media_type="application/pdf")



@app.get("/api/sse/{tid}")
def sse(tid: str):
    def event_generator():
        progress_file = f"{PROGRESS_DIR}/{tid}.json"
        last_sent = None
        
        while True:
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, encoding="utf-8") as f:
                        data = json.load(f)
                    current = json.dumps(data, ensure_ascii=False)
                    if current != last_sent:
                        yield f"data: {current}\n\n"
                        last_sent = current
                    if data.get("done"):
                        break
                except Exception:
                    pass
            time.sleep(0.4)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



@app.get("/download/{task_id}")
async def download_file(task_id: str):
    file_path = f"{OUT_DIR}/{task_id}.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在或尚未完成处理")
    
    # 检查进度文件是否存在且 done=True
    progress_path = f"{PROGRESS_DIR}/{task_id}.json"
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)
        if not progress.get("done"):
            raise HTTPException(status_code=400, detail="处理尚未完成")
    
    return FileResponse(
        path=file_path,
        filename=f"ocr_result_{task_id}.pdf",
        media_type="application/pdf"
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8038, reload=False)