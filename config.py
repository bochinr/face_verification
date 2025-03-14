# 配置文件
import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径配置
MODEL_PATHS = {
    "haar_cascade": os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml"),
    "feature_extractor": os.path.join(BASE_DIR, "models", "nn4.small2.v1.t7")
}

# 算法参数配置
THRESHOLD = 0.6              # 相似度阈值
FACE_DETECTION_SCALE = 1.1   # 人脸检测缩放比例
MIN_NEIGHBORS = 5            # 最小邻居数
MIN_SIZE = (30, 30)          # 最小人脸尺寸