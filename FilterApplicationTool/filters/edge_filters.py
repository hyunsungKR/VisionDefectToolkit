from .base_filter import BaseFilter
import cv2
import numpy as np

class LaplacianFilter(BaseFilter):
    def __init__(self):
        super().__init__("Laplacian")
        self.params = {"ksize": 3}
    
    def apply(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = min(31, max(3, int(3 + 2 * intensity * 2)))
        if ksize % 2 == 0:
            ksize += 1
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        filtered = cv2.convertScaleAbs(laplacian)
        return self.blend_with_original(image, filtered, intensity)

class SobelFilter(BaseFilter):
    def __init__(self, direction='x'):
        super().__init__(f"Sobel {direction.upper()}")
        self.direction = direction
        self.params = {"ksize": 3}
    
    def apply(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = min(31, max(3, int(3 + 2 * intensity * 2)))
        if ksize % 2 == 0:
            ksize += 1
        
        if self.direction == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        filtered = cv2.convertScaleAbs(sobel)
        return self.blend_with_original(image, filtered, intensity)

class ScharrFilter(BaseFilter):
    def __init__(self, direction='x'):
        super().__init__(f"Scharr {direction.upper()}")
        self.direction = direction
    
    def apply(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.direction == 'x':
            scharr = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        else:
            scharr = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        filtered = cv2.convertScaleAbs(scharr)
        return self.blend_with_original(image, filtered, intensity)

class PrewittFilter(BaseFilter):
    def __init__(self):
        super().__init__("Prewitt")
    
    def apply(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) * intensity
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) * intensity
        
        processed_x = cv2.filter2D(gray, cv2.CV_32F, kernelx)
        processed_y = cv2.filter2D(gray, cv2.CV_32F, kernely)
        filtered = cv2.magnitude(processed_x, processed_y)
        filtered = cv2.convertScaleAbs(filtered)
        return self.blend_with_original(image, filtered, intensity)

class CannyFilter(BaseFilter):
    def __init__(self):
        super().__init__("Canny Edge")
        self.params = {
            "threshold1": 50,
            "threshold2": 150
        }
    
    def apply(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold1 = int(self.params["threshold1"] * intensity)
        threshold2 = int(self.params["threshold2"] * intensity)
        filtered = cv2.Canny(gray, threshold1, threshold2)
        return self.blend_with_original(image, filtered, intensity) 