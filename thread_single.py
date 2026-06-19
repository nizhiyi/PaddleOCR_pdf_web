from paddleocr import PaddleOCR
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import paddle
class PaddleOCRModelManager(ThreadPoolExecutor):
    def get_device(self):
        try:
            if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                self.logger.info("检测到可用 GPU，多线程模式下启用 GPU 加速")
                return "gpu:0"
        except Exception as e:
            self.logger.warning(f"GPU 检测失败: {e}")
        self.logger.info("使用 CPU 进行 OCR")
        return "cpu"


    def __init__(self, current_app, **kwargs):
        # 增加线程池大小并设置线程名称
        super(PaddleOCRModelManager, self).__init__(max_workers=1, thread_name_prefix="paddle_ocr_", **kwargs)
        os.environ["PADDLE_PDX_CACHE_HOME"] = "./module"
        self.logger = current_app.logger
        self.logger.info("初始化PaddleOCR模型管理器...")
        try:
            self.paddleocr = PaddleOCR(
                text_detection_model_name="PP-OCRv6_small_det",
                text_recognition_model_name="PP-OCRv6_small_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=self.get_device(),
                text_det_limit_type="min",
                text_det_limit_side_len=64,
                text_det_thresh=0.2,
                text_det_box_thresh=0.5,
                text_det_unclip_ratio=1.5,
                text_rec_score_thresh=0
            )
            self.logger.info("PaddleOCR模型初始化成功")
        except Exception as e:
            self.logger.error(f"PaddleOCR模型初始化失败: {str(e)}")
            raise
        self.app = current_app
        self.active_tasks = 0


    def submit_ocr(self, **kwargs):
        self.active_tasks += 1
        self.logger.info(f"提交OCR任务，当前活跃任务数: {self.active_tasks}")
        try:
            # 添加超时参数，防止单个任务阻塞过长时间
            future = self.submit(self.infer, **kwargs)
            result = future.result(timeout=600)  # 设置10分钟超时
            return result
        except TimeoutError:
            self.logger.error(f"OCR任务执行超时")
            raise TimeoutError("OCR处理超时，请检查输入图像质量和服务器负载")
        except Exception as e:
            self.logger.error(f"OCR任务执行异常: {str(e)}")
            raise
        finally:
            self.active_tasks -= 1
            self.logger.info(f"OCR任务完成，当前活跃任务数: {self.active_tasks}")


    def infer(self, **kwargs):
        start_time = time.time()
        input_path = kwargs.get('input', '')
        json_path = kwargs.get('json_path', '')
        self.logger.info(f"开始OCR推理，输入: {input_path}")
        try:
            result_str = self.paddleocr.predict(input_path)
            processing_time = time.time() - start_time
            self.logger.info(f"OCR推理完成，处理时间: {processing_time:.2f}秒")
            result = self.print_order_no(result_str,json_path)
            self.logger.info(f"OCR推理结果: {result}")
            return result, result_str
        except Exception as e:
            self.logger.error(f"OCR推理异常: {str(e)}")
            raise


    def print_order_no(self, result,json_path):
        res_str = ""
        try:
            for res in result:
                res.save_to_json(json_path)         
            self.logger.info(f"OCR结果处理完成，识别文本数: {sum(len(res['rec_texts']) for res in result)}")
            return res_str
        except Exception as e:
            self.logger.error(f"OCR结果处理异常: {str(e)}")
            raise