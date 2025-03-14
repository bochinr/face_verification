# 人脸处理模块
import cv2
import numpy as np
from config import MODEL_PATHS, FACE_DETECTION_SCALE, MIN_NEIGHBORS, MIN_SIZE


class FaceProcessor:
    def __init__(self):
        # 初始化人脸检测器
        try:
            self.face_cascade = cv2.CascadeClassifier(MODEL_PATHS["haar_cascade"])
        except Exception as e:
            print(f"Error loading Haar cascade: {e}")
            print("Falling back to OpenCV's default path.")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_faces(self, image):
        """检测图像中的人脸"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION_SCALE,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )
        return faces

    @staticmethod
    def align_face(image, face_box):
        """对齐人脸并调整尺寸"""
        x, y, w, h = face_box
        face_roi = image[y:y + h, x:x + w]
        return cv2.resize(face_roi, (96, 96))

    @staticmethod
    def annotate_image(image, faces):
        """在图像上标注人脸框和坐标"""
        annotated = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, f"({x}, {y})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated