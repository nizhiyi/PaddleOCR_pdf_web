from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse,FileResponse
import os
import uuid
import subprocess
import json
import time

BASE = "."
PDF_DIR = os.path.abspath("input_pdfs")
OUT_DIR = os.path.abspath("output_pdfs")
PROGRESS_DIR = os.path.abspath("progress")

for d in [PDF_DIR, OUT_DIR, PROGRESS_DIR]:
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="PaddleOCR PDF 可搜索化服务")

# ================== 前端单页（传统按钮上传风格） ==================
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

  <!-- Element Plus Icons（只有 JS，没有 CSS） -->
  <script src="https://cdn.jsdelivr.net/npm/@element-plus/icons-vue@2.3.2"></script>

  <style>
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background:#f5f7fa;
      margin:0;
      padding:20px;
    }
    #app {
      max-width:760px;
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
  </style>
</head>

<body>
<div id="app">
  <div class="header">
    <h2>PaddleOCR PDF</h2>
    <p style="color:#666;">上传 PDF → OCR 识别 → 生成可搜索 PDF</p>
  </div>

  <!-- 上传区 -->
  <div class="upload-area" v-if="!taskId">
    <div style="margin-bottom: 20px;">
      <el-checkbox v-model="isVisible" label="生成带原图背景的可搜索 PDF" size="large" />
    </div>
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

  <!-- 进度区 -->
  <div v-else class="progress-box">
    <el-card shadow="hover">
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

    <!-- 结果 -->
    <div class="result-box" v-if="done">
      <el-result
        icon="success"
        title="处理完成"
        sub-title="已生成可搜索 PDF"
      >
        <template #extra>
          <el-button type="primary" size="large" @click="download">
            <el-icon><Download /></el-icon>
            下载 PDF
          </el-button>
          <el-button size="large" @click="reset">
            再处理一个
          </el-button>
        </template>
      </el-result>
    </div>
  </div>
</div>

<script>
const { createApp, ref } = Vue

const app = createApp({
  setup() {
    const taskId = ref('')
    const percent = ref(0)
    const stage = ref('')
    const done = ref(false)
    const isVisible = ref(true)

    const handleFileChange = (fileItem) => {
      console.log('fileItem:', fileItem)
      if (fileItem?.raw) {
        uploadFile(fileItem.raw)
      }
    }

    const uploadFile = async (file) => {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('visible', isVisible.value)

      const res = await fetch('/api/ocr', {
        method: 'POST',
        body: fd
      })
      const data = await res.json()
      taskId.value = data.task_id
      listenSSE()
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
      taskId.value = ''
      percent.value = 0
      stage.value = ''
      done.value = false
    }

    const listenSSE = () => {
      const es = new EventSource(`/api/sse/${taskId.value}`)
      es.onmessage = e => {
        const d = JSON.parse(e.data)
        percent.value = d.percent || 0
        stage.value = d.stage || ''
        if (d.done) {
          done.value = true
          es.close()
        }
      }
    }

    return { taskId, percent, stage, done, isVisible, handleFileChange, download, reset }
  }
})

/* 🔥 必须：注册 Element Plus */
app.use(ElementPlus)

/* 🔥 必须：注册 Icons */
Object.entries(ElementPlusIconsVue).forEach(([name, comp]) => {
  app.component(name, comp)
})

app.mount('#app')
</script>

</body>
</html>
    """

@app.post("/api/ocr")
async def ocr(file: UploadFile, visible: bool = Form(True)):
    tid = str(uuid.uuid4())
    pdf_path = os.path.join(PDF_DIR, f"{tid}.pdf")
    out_path = os.path.join(OUT_DIR, f"{tid}.pdf")

    try:
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        print(f"写入上传文件失败: {e}")
        raise HTTPException(status_code=500, detail="文件上传保存失败")

    cmd = [
        "python",
        "pdf_ocr_select.py",
        "--pdf", pdf_path,
        "--out", out_path,
        "--task_id", tid
    ]
    if visible:
        cmd.append("--visible")

    print(f"启动 OCR 进程: {' '.join(cmd)}")
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"启动子进程失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动处理进程失败: {e}")

    return {"task_id": tid}


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
    
    # 可选：检查进度文件是否存在且 done=True
    progress_path = f"{PROGRESS_DIR}/{task_id}.json"
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)
        if not progress.get("done"):
            raise HTTPException(status_code=400, detail="处理尚未完成")
    
    return FileResponse(
        path=file_path,
        filename=f"ocr_result_{task_id}.pdf",  # 下载时显示友好文件名
        media_type="application/pdf"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8038, reload=False)