from abc import ABC, abstractmethod
import cv2
import numpy as np

class BaseFilter(ABC):
    def __init__(self, name):
        self.name = name
        self.intensity = 1.0
        self.params = {}
    
    @abstractmethod
    def apply(self, image, intensity=1.0):
        pass
    
    def blend_with_original(self, original, filtered, intensity):
        """원본 이미지와 필터링된 이미지를 블렌딩"""
        if len(filtered.shape) == 2:  # 그레이스케일 이미지인 경우
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(original, 1 - intensity, filtered, intensity, 0) 