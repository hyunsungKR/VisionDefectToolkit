from .base_filter import BaseFilter
import cv2
import numpy as np

class BandpassFilter(BaseFilter):
    def __init__(self):
        super().__init__("Bandpass Filter")
        self.params = {
            "radius_outer": 60,
            "radius_inner": 10
        }
    
    def apply(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        radius_outer = int(self.params["radius_outer"] * intensity)
        radius_inner = int(self.params["radius_inner"] * intensity)
        
        mask = np.zeros((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), radius_outer, (1, 1), -1)
        cv2.circle(mask, (ccol, crow), radius_inner, (0, 0), -1)
        
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        filtered = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return self.blend_with_original(image, filtered, intensity)

class GaborFilter(BaseFilter):
    def __init__(self):
        super().__init__("Gabor Filter")
        self.params = {
            "ksize": 21,
            "sigma": 8.0,
            "theta": np.pi/4,
            "lambda": 10.0,
            "gamma": 0.5
        }
    
    def apply(self, image, intensity=1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = min(31, int(self.params["ksize"] * intensity))
        if ksize % 2 == 0:
            ksize += 1
            
        gabor_kernel = cv2.getGaborKernel(
            (ksize, ksize),
            self.params["sigma"],
            self.params["theta"],
            self.params["lambda"] * intensity,
            self.params["gamma"]
        )
        
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
        filtered = cv2.convertScaleAbs(filtered)
        return self.blend_with_original(image, filtered, intensity) 