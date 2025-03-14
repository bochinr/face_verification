# 主程序
import tkinter as tk
from config import THRESHOLD
from face_processing import FaceProcessor
from feature_extractor import FeatureExtractor
from gui import FaceVerificationGUI

def main():
    # 初始化处理模块
    processor = FaceProcessor()
    extractor = FeatureExtractor()
    
    # 创建GUI
    root = tk.Tk()
    app = FaceVerificationGUI(root, processor, extractor)
    root.mainloop()

if __name__ == "__main__":
    main()