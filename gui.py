# gui.py

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from config import THRESHOLD  # 导入 THRESHOLD


class FaceVerificationGUI:
    def __init__(self, master, processor, extractor):
        self.master = master
        self.processor = processor
        self.extractor = extractor

        # 初始化状态
        self.image_paths = [None, None]
        self.image_labels = [None, None]

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        """创建界面组件"""
        self.master.title("人脸验证系统")

        # 图片1上传区域
        frame1 = tk.Frame(self.master)
        frame1.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame1, text="图片1").pack()
        self.img_label1 = tk.Label(frame1)
        self.img_label1.pack()
        tk.Button(frame1, text="上传图片",
                  command=lambda: self.upload_image(0)).pack()

        # 图片2上传区域
        frame2 = tk.Frame(self.master)
        frame2.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame2, text="图片2").pack()
        self.img_label2 = tk.Label(frame2)
        self.img_label2.pack()
        tk.Button(frame2, text="上传图片",
                  command=lambda: self.upload_image(1)).pack()

        # 验证按钮
        tk.Button(self.master, text="开始验证",
                  command=self.run_verification).pack(pady=20)

    def upload_image(self, index):
        """处理图片上传"""
        path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg;*.jpeg;*.png")])
        if path:
            self.image_paths[index] = path
            self.show_thumbnail(index, path)

    def show_thumbnail(self, index, path):
        """显示缩略图"""
        img = Image.open(path)
        img.thumbnail((150, 150))
        img_tk = ImageTk.PhotoImage(img)

        if index == 0:
            self.img_label1.config(image=img_tk)
            self.img_label1.image = img_tk
        else:
            self.img_label2.config(image=img_tk)
            self.img_label2.image = img_tk

    def run_verification(self):
        """执行验证流程"""
        # 输入检查
        if None in self.image_paths:
            self.show_error("请先上传两张图片")
            return

        # 读取并检测图片
        try:
            img1 = cv2.imread(self.image_paths[0])
            img2 = cv2.imread(self.image_paths[1])

            faces1 = self.processor.detect_faces(img1)
            faces2 = self.processor.detect_faces(img2)

            # 显示检测结果
            self.show_detection_result(img1, faces1, "图片1检测结果")
            self.show_detection_result(img2, faces2, "图片2检测结果")

            # 验证逻辑
            if len(faces1) == 0 or len(faces2) == 0:
                self.show_error("至少一张图片未检测到人脸")
                return

            # 提取特征
            aligned1 = self.processor.align_face(img1, faces1[0])
            aligned2 = self.processor.align_face(img2, faces2[0])

            vec1 = self.extractor.get_features(aligned1)
            vec2 = self.extractor.get_features(aligned2)

            # 显示结果
            similarity = self.extractor.compare_features(vec1, vec2)
            result = "同一人" if similarity > THRESHOLD else "不同人"  # 使用 THRESHOLD
            self.show_result(f"相似度: {similarity:.2f}\n验证结果: {result}")

        except Exception as e:
            self.show_error(f"处理出错: {str(e)}")

    def show_detection_result(self, image, faces, title):
        """显示带标注的图片"""
        annotated = self.processor.annotate_image(image, faces)
        cv2.imshow(title, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_result(self, message):
        """显示验证结果"""
        result_window = tk.Toplevel()
        result_window.title("验证结果")
        tk.Label(result_window, text=message, font=("Arial", 14)).pack(padx=20, pady=20)

    def show_error(self, message):
        """显示错误提示"""
        error_window = tk.Toplevel()
        error_window.title("错误")
        tk.Label(error_window, text=message, fg="red").pack(padx=20, pady=20)