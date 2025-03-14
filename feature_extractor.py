# 特征提取模块
import cv2
from config import MODEL_PATHS
from sklearn.metrics.pairwise import cosine_similarity

class FeatureExtractor:
    def __init__(self):
        # 加载特征提取模型
        self.net = cv2.dnn.readNetFromTorch(MODEL_PATHS["feature_extractor"])
        
    def get_features(self, image):
        """从对齐的人脸图像提取特征向量"""
        blob = cv2.dnn.blobFromImage(
            image, 
            1.0/255, 
            (96, 96), 
            (0, 0, 0), 
            swapRB=True, 
            crop=False
        )
        self.net.setInput(blob)
        return self.net.forward().flatten()
    
    @staticmethod
    def compare_features(vec1, vec2):
        """计算两个特征向量的余弦相似度"""
        return cosine_similarity([vec1], [vec2])[0][0]