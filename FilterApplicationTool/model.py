# model.py

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog
import os
from filters.edge_filters import (
    LaplacianFilter, SobelFilter, ScharrFilter,
    PrewittFilter, CannyFilter
)
from filters.frequency_filters import BandpassFilter, GaborFilter

class FilterApplicationModel:
    def __init__(self):
        self.original_images = []
        self.filtered_images = []
        self.main_image = None
        self.main_filter = "Original"
        self.filter_intensity = 1.0
        
        # 필터 객체들 초기화
        self.filters = {
            "Original": None,
            "Bandpass Filter": BandpassFilter(),
            "Gabor Filter": GaborFilter(),
            "Laplacian": LaplacianFilter(),
            "Sobel X": SobelFilter('x'),
            "Sobel Y": SobelFilter('y'),
            "Scharr X": ScharrFilter('x'),
            "Scharr Y": ScharrFilter('y'),
            "Prewitt": PrewittFilter(),
            "Canny Edge": CannyFilter()
        }
        
        self.filter_intensities = {name: 1.0 for name in self.filters}
    
    def load_images(self, from_directory=False):
        """Load images either individually or from a directory."""
        if from_directory:
            directory = QFileDialog.getExistingDirectory(None, "Select Directory")
            if not directory:
                return []

            file_paths = []
            for root, _, files in os.walk(directory):
                for filename in files:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        file_paths.append(os.path.join(root, filename))
        else:
            file_paths, _ = QFileDialog.getOpenFileNames(None, "Load Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
            if not file_paths:
                return []

        loaded_images = []
        for path in file_paths:
            img = self.load_image_with_pil(path)
            if img is not None:
                loaded_images.append(img)
            if len(loaded_images) >= 1:  # Limit to 1 original image for 10 filters
                break

        if not loaded_images:
            return []

        self.original_images = loaded_images
        self.filtered_images = []
        for idx, img in enumerate(self.original_images):
            for filter_name in self.filters.keys():
                filtered = self.apply_filter(filter_name, img, self.filter_intensity)
                self.filtered_images.append((idx, filter_name, filtered))
                if len(self.filtered_images) >= 10:
                    break
            if len(self.filtered_images) >= 10:
                break

        return [(self.convert_cv_qt(img), filter_name) for (_, filter_name, img) in self.filtered_images]

    def load_image_with_pil(self, file_path):
        """Load image using PIL and convert to OpenCV format."""
        try:
            pil_image = Image.open(file_path).convert("RGB")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None

    def set_main_image_by_filter(self, filter_name):
        """Set the main image based on the filter name."""
        for orig_idx, fname, img in self.filtered_images:
            if fname == filter_name:
                self.main_image = self.original_images[orig_idx]
                self.main_filter = fname
                self.filter_intensity = self.filter_intensities[fname]
                break

    def update_filter_intensity(self, intensity):
        """Update the intensity for the current main filter and reapply it."""
        self.filter_intensity = intensity
        self.filter_intensities[self.main_filter] = intensity
        
        if self.main_image is not None:
            filtered = self.apply_filter(self.main_filter, self.main_image, intensity)
            if filtered is not None:
                for idx, (orig_idx, fname, _) in enumerate(self.filtered_images):
                    if orig_idx == self.original_images.index(self.main_image) and fname == self.main_filter:
                        self.filtered_images[idx] = (orig_idx, fname, filtered)
                        break
                return self.convert_cv_qt(filtered)
        return None

    def apply_main_filter(self):
        """Apply the main filter to the main image and return QPixmap."""
        if self.main_image is not None and self.main_filter in self.filters:
            filtered_image = self.apply_filter(self.main_filter, self.main_image, self.filter_intensity)
            return self.convert_cv_qt(filtered_image)
        return None

    def apply_filter(self, filter_name, image, intensity=1.0):
        """Apply a specific filter to the image."""
        if filter_name == "Original":
            return image.copy()
            
        filter_obj = self.filters.get(filter_name)
        if filter_obj:
            return filter_obj.apply(image, intensity)
        return image.copy()

    def convert_cv_qt(self, cv_img):
        """Convert OpenCV image to QPixmap."""
        if cv_img.ndim == 2:
            # Grayscale image
            q_image = QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], cv_img.strides[0], QImage.Format.Format_Grayscale8)
        else:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)
